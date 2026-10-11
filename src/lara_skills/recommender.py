"""Local text-to-skill recommendation API; descriptions are encoded in memory and never stored."""

import io
import json
from pathlib import Path

import numpy as np
import torch

from .encoders import make_encoder
from .hierarchy import load_hierarchy
from .models import MODEL_NAMES, make_ranker
from .occupations import load_occupations
from .patent_embeddings import query_text
from .pipeline_io import digest
from .ranking import baseline_scores, torch_adjacency
from .recommender_data import load_market
from .training_inputs import verify_files

BASELINES = {"popularity", "semantic", "diffusion"}


class RecommendationEngine:
    """Score every dictionary skill for a new description with a trained bundle or a baseline."""

    def __init__(self, root, model_dir, model=None, device="cpu", candidate_dir=None,
                 hierarchy_dir=None, occupation_dir=None):
        self.root = Path(root)
        directory = self.root / model_dir
        verify_files(directory, ["config.json", "encoder_config.json", "diffusion_selection.json"])
        self.config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        self.encoder_config = json.loads((directory / "encoder_config.json").read_text(encoding="utf-8"))
        if self.config.get("task_mode", "work_types") != self.encoder_config.get("task_mode", "work_types"):
            raise ValueError("Model and text encoder task modes differ")
        self.model_name = model or self.config["default_engine"]
        self.device = device
        self.candidate_dir = candidate_dir or self.config["candidate_dir"]
        self.candidates, self.features, self.adjacency, self.frequency = load_market(self.root, self.candidate_dir)
        reference = json.loads((self.root / self.candidate_dir / "feature_reference.json").read_text(encoding="utf-8"))
        feature_path = self.root / reference["path_relative_to_project_root"]
        feature_manifest = json.loads((feature_path.parent / "manifest.json").read_text(encoding="utf-8"))
        for key in ["model_id", "revision"]:
            if feature_manifest["config"][key] != self.encoder_config[key]:
                raise ValueError("Market and query text encoders are incompatible")
        self.alpha = json.loads((directory / "diffusion_selection.json").read_text(encoding="utf-8"))["alpha"]
        self.model = None
        if self.model_name in MODEL_NAMES:
            verify_files(directory, [f"{self.model_name}.pt"])
            checkpoint = torch.load(io.BytesIO((directory / f"{self.model_name}.pt").read_bytes()),
                                    map_location="cpu", weights_only=True)
            model_manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            chosen_hierarchy = hierarchy_dir or self.config.get("hierarchy_dir")
            if chosen_hierarchy is None:
                raise ValueError("A hierarchy bundle is required")
            hierarchy = load_hierarchy(self.root, chosen_hierarchy, self.candidate_dir)
            if hierarchy_dir is None and digest(self.root / chosen_hierarchy / "manifest.json") != model_manifest["input_manifests"]["hierarchy_dir"]:
                raise ValueError("Training hierarchy version has changed")
            chosen_occupation = occupation_dir or self.config.get("occupation_dir")
            if chosen_occupation is None:
                raise ValueError("An occupation bundle is required")
            occupations = load_occupations(self.root, chosen_occupation, self.candidate_dir)
            encoding = json.loads((self.root / chosen_occupation / "encoder_config.json").read_text(encoding="utf-8"))
            if any(encoding[key] != self.encoder_config[key] for key in ["model_id", "revision"]):
                raise ValueError("Occupation and query encoders differ")
            if occupation_dir is None and digest(self.root / chosen_occupation / "manifest.json") != model_manifest["input_manifests"]["occupation_dir"]:
                raise ValueError("Training occupation version has changed")
            self.model = make_ranker(self.model_name, checkpoint["input_dim"], checkpoint["perspective_count"],
                                     checkpoint["config"], hierarchy, occupations).to(device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
            self.x = torch.as_tensor(self.features, device=device)
            self.a = torch_adjacency(self.adjacency, device)
            self.f = torch.as_tensor(self.frequency, device=device)
        elif self.model_name not in BASELINES:
            raise ValueError(f"Unknown recommender model {self.model_name!r}")
        self.encoder = None

    def score_embeddings(self, vectors, perspective_ids):
        vectors = np.asarray(vectors, dtype=np.float32)
        perspective_ids = np.asarray(perspective_ids, dtype=np.int64)
        if vectors.ndim != 2 or vectors.shape[1] != self.features.shape[1] or len(vectors) != len(perspective_ids):
            raise ValueError("Invalid query embedding dimensions")
        if not np.isfinite(vectors).all() or np.any(perspective_ids < 0) or np.any(perspective_ids >= len(self.encoder_config["perspectives"])):
            raise ValueError("Invalid query features or perspective")
        if self.model is None:
            scores = baseline_scores(self.model_name, vectors, self.features,
                                     self.candidates.posting_count.to_numpy(), self.adjacency, self.alpha)
        else:
            with torch.no_grad():
                scores = self.model.score_all(torch.as_tensor(vectors, device=self.device),
                                               torch.as_tensor(perspective_ids, device=self.device), self.x, self.a, self.f,
                                               self.config["score_chunk_size"]).cpu().numpy()
        if scores.shape != (len(vectors), len(self.candidates)) or not np.isfinite(scores).all():
            raise ValueError("Not every candidate received a finite score")
        return scores

    def encode(self, description):
        texts = [query_text(description, p, self.encoder_config) for p in self.encoder_config["perspectives"]]
        if self.encoder is None:
            self.encoder = make_encoder(self.encoder_config, device=self.device)
        vectors, _ = self.encoder.encode(texts)
        return vectors

    def _check_top_k(self, top_k):
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= len(self.candidates):
            raise ValueError("Invalid number of recommendations")

    def recommend(self, description, top_k=5):
        """Skill names only: a list for single-task bundles, a dict per perspective otherwise."""
        self._check_top_k(top_k)
        scores = self.score_embeddings(self.encode(description), np.arange(len(self.encoder_config["perspectives"])))
        results = {perspective: self.candidates.iloc[np.argsort(-scores[i], kind="stable")[:top_k]].skill_name.tolist()
                   for i, perspective in enumerate(self.encoder_config["perspectives"])}
        return results["skills"] if self.encoder_config.get("task_mode") == "single" else results

    def recommend_table(self, description, top_k=5):
        """Ranked rows with codes, names and scores for the single-task interface."""
        self._check_top_k(top_k)
        if self.encoder_config.get("task_mode") != "single":
            raise ValueError("recommend_table expects a single-task bundle")
        scores = self.score_embeddings(self.encode(description), np.zeros(1, dtype=np.int64))[0]
        rows = []
        for rank, cid in enumerate(np.argsort(-scores, kind="stable")[:top_k], 1):
            candidate = self.candidates.iloc[int(cid)]
            rows.append({"rank": rank, "skill_code": candidate.skill_code, "skill_name": candidate.skill_name,
                         "score": float(scores[cid]), "posting_count": int(candidate.posting_count)})
        return rows
