from fastapi import APIRouter, HTTPException, Request

from gamelens.db import DatabaseConnection
from gamelens.prediction.event import segment_events
from gamelens.prediction.inference import PyTorchInferencer
from gamelens.prediction.model import CapturesClassification

router = APIRouter(prefix="/api/v1/classifier", tags=["classifier"])


@router.get("/", response_model=CapturesClassification)
async def classify_event(request: Request, run_id: str, chunk_size: int = 100):
    if chunk_size < 1:
        raise HTTPException(status_code=400, detail="chunk_size must be 1 or greater.")

    try:
        async with DatabaseConnection.get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT
                    image_data,capture_id
                    FROM
                    raw_capture
                    WHERE
                    run_id = %s
                    ORDER BY capture_index ASC;
                    """,
                    (run_id,),
                )
                # Fetch rows in chuncks, since we could exhust
                # our memory if we load a long run
                inferencer: PyTorchInferencer = request.state.inferencer

                classifications = []
                length = 0
                print(f"Processing {cur.rowcount} images from DB...")
                while True:
                    sorted_captures = await cur.fetchmany(chunk_size)

                    if not sorted_captures and length == 0:
                        raise HTTPException(
                            status_code=404,
                            detail=f"No images found for run_id: {run_id}",
                        )

                    length += len(sorted_captures)
                    if not sorted_captures:
                        print(f"Finished processing {length} images.")
                        break

                    for capture in sorted_captures:
                        image_bytes = bytes(capture[0])
                        capture_id = capture[1]
                        try:
                            result = inferencer.process_image(image_bytes, capture_id)
                            classifications.append(result)
                        except Exception as e:
                            print(f"Error processing image {capture_id}: {e}")
                            raise HTTPException(
                                status_code=422,
                                detail=f"Failed to process image {capture_id}.",
                            )
                    print(f"Processed {length}/{cur.rowcount} images.")

        interval_captures = segment_events(classifications)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to classify images: {e}",
        )
    return {"predictions": interval_captures}
