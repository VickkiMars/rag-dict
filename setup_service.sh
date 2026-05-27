#!/bin/bash

# setup_service.sh - Automate systemd installation for RAG Dictionary
set -e

echo "=================================================="
echo "   LexiSeek RAG — Host Service Installer (systemd)"
echo "=================================================="

# 1. Detect Username and Path
USER_NAME=$(whoami)
WORK_DIR=$(pwd)

echo "🔍 Detected User: $USER_NAME"
echo "🔍 Detected Working Directory: $WORK_DIR"

# 2. Locate Virtual Environment containing uvicorn
VENV_PATH=""
for dir in venv .venv env; do
    if [ -f "$WORK_DIR/$dir/bin/uvicorn" ]; then
        VENV_PATH="$WORK_DIR/$dir"
        break
    fi
done

# Search 1 level deep if not found in standard directories
if [ -z "$VENV_PATH" ]; then
    found_path=$(find . -maxdepth 3 -name "uvicorn" -path "*/bin/uvicorn" | head -n 1)
    if [ -n "$found_path" ]; then
        # Clean relative prefix and convert to absolute path
        clean_rel=$(echo "$found_path" | sed 's|^\./||')
        VENV_PATH="$WORK_DIR/$(dirname $(dirname "$clean_rel"))"
    fi
fi

# Fallback to the user's custom codebase venv
if [ -z "$VENV_PATH" ]; then
    if [ -f "/home/kami/Desktop/codebase/main/bin/uvicorn" ]; then
        VENV_PATH="/home/kami/Desktop/codebase/main"
    fi
fi

if [ -z "$VENV_PATH" ]; then
    echo "❌ Error: Could not locate a python virtual environment containing 'bin/uvicorn' in this workspace."
    echo "Please activate your virtual environment or install the dependencies first:"
    echo "  source <your-venv>/bin/activate && pip install -r requirements.txt"
    exit 1
fi

echo "🔍 Detected Virtual Environment: $VENV_PATH"

# 3. Create the Systemd Service File
SERVICE_FILE="/etc/systemd/system/rag-dict.service"
echo "📝 Generating systemd service file at $SERVICE_FILE..."

sudo bash -c "cat > $SERVICE_FILE" <<EOF
[Unit]
Description=LexiSeek RAG Dictionary Service
After=network.target redis-server.service

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$WORK_DIR
ExecStart=$VENV_PATH/bin/uvicorn src.serve:app --host 127.0.0.1 --port 8000
Restart=always
Environment=PYTHONPATH=$WORK_DIR
Environment=REDIS_HOST=127.0.0.1
Environment=REDIS_PORT=6379
Environment=DICT_JSON=$WORK_DIR/data/dictionary.json
Environment=FAISS_PATH=$WORK_DIR/data/embeddings.faiss

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Service file created successfully."

# 4. Reload, Enable and Start the service
echo "⚙️ Reloading systemd daemon..."
sudo systemctl daemon-reload

echo "⚙️ Enabling rag-dict.service..."
sudo systemctl enable rag-dict.service

echo "⚙️ Starting (or restarting) rag-dict.service..."
sudo systemctl restart rag-dict.service

echo "=================================================="
echo "🎉 Setup Completed Successfully!"
echo "=================================================="
echo "Check service status:  systemctl status rag-dict.service"
echo "Test the lookup CLI:  define love"
echo "=================================================="
