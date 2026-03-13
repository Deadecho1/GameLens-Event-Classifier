import contextlib
import gc

from fastapi import FastAPI

from gamelens.db import DatabaseConnection
from gamelens.extraction.choice import router as choice_router
from gamelens.prediction.config import CLASSES, MODEL_PATH
from gamelens.prediction.inference import PyTorchInferencer
from gamelens.routes.classifier import router as classifier_router
from gamelens.util import init_db


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    inferencer = PyTorchInferencer(MODEL_PATH, CLASSES)

    yield {"inferencer": inferencer}

    print("Shutting down FastAPI application...")
    await DatabaseConnection.close()
    del inferencer
    gc.collect()


app = FastAPI(lifespan=lifespan)
app.include_router(classifier_router)
app.include_router(choice_router)
