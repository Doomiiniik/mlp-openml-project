# Letter Classifier API

A small REST API that serves a PyTorch MLP classifying capital letters (A–Z)
from 16 numeric features. Built to practice deploying an ML model:
FastAPI, Docker, Nginx and HTTPS.

Live demo: https://api.doomiiniik.dev/

Model training lives in a separate repo: [mlp-openml-project](https://github.com/Doomiiniik/mlp-openml-project)

## Model

- PyTorch MLP: 16 inputs → 256 → 128 → 26 classes (ReLU, dropout)
- Dataset: [OpenML Letter Recognition (ID 6)](https://www.openml.org/d/6), 16 numeric features
  computed from images of printed capital letters
- Test accuracy: 96.0% (3,734 test samples)

The API takes the 16 precomputed features, not an image.

## API

`POST /v1/predict`

Request:
```json
{ "features": [2, 8, 3, 5, 1, 8, 13, 0, 6, 6, 10, 8, 0, 8, 0, 8] }
```

Response:
```json
{ "predicted_class": "19", "probabilities": { "0": 0.0, "1": 0.0, "...": "..." } }
```

- `features` are raw dataset values (integers 0–15). The API applies the same
  standardization as in training (mean and std stored in `configs/scaler.json`).
- `predicted_class` is the class index (0–25, 0 = A, so 19 = T). The frontend maps it to a letter.
- A request with a number of features other than 16 is rejected with HTTP 422.

More inputs to try: `[5,12,3,7,2,10,5,5,4,13,3,9,2,8,4,10]` (I), `[4,11,6,8,6,10,6,2,6,10,3,7,2,8,3,9]` (D).

Other endpoints: `GET /docs` (Swagger UI), `GET /metrics` (Prometheus metrics).

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Or with Docker (API + Prometheus + Grafana):

```bash
docker compose up --build
```

API docs: http://localhost:8000/docs

## Deployment

Client → Nginx (HTTPS) → FastAPI on 127.0.0.1:8000 → model

- Ubuntu server, Nginx as reverse proxy and HTTPS termination
- Certificate from Let's Encrypt (Certbot, automatic renewal)
- Nginx serves the static frontend and proxies `/v1/`, `/docs`, `/openapi.json` to the backend

## Project structure

```
app/main.py              app setup, middleware, routes
app/api/v1/endpoints/    /predict endpoint
app/schemas/             request/response validation (Pydantic)
app/models/              model architecture, loading, inference
app/middleware/          logging, error handling, metrics
frontend/                static demo page
models/, configs/        trained weights, model config, scaler parameters
```

## Limitations

- Input is a feature vector, not an image, so the demo is not very intuitive
- No automated tests yet
- No CI/CD, deployment is manual
