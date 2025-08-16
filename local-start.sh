#!/bin/bash
set -e

cleanup() {
  echo "Stopping Docker services..."
  docker compose down
}
trap cleanup EXIT

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Starting Docker services..."
docker compose up -d

# Wait for Elasticsearch to be ready on port 9200
echo "Waiting for Elasticsearch to start..."
until curl -s http://localhost:9200 >/dev/null; do
  sleep 2
done
echo "Elasticsearch is ready!"

echo "Running text embedding script..."
python text_embedding.py

echo "Starting FastAPI server with Uvicorn..."
# Pipe Uvicorn logs to a temporary file
LOG_FILE=$(mktemp)
uvicorn main:app --reload > "$LOG_FILE" 2>&1 &
UVICORN_PID=$!

# Wait until Uvicorn log contains "Application startup complete."
echo "Waiting for FastAPI to start..."
until grep -q "Application startup complete." "$LOG_FILE"; do
  sleep 1
done
sleep 2
echo "FastAPI is ready!"

echo "Starting Streamlit app..."
streamlit run app.py &
STREAMLIT_PID=$!

# Forward Uvicorn logs to console
tail -f "$LOG_FILE" &

# Keep script running until both background processes stop
wait $UVICORN_PID $STREAMLIT_PID
