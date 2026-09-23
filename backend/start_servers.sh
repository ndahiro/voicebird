#!/bin/bash
# Start MySQL, ASR and Translation servers for VoiceBird

echo "Starting VoiceBird Local Servers..."

# Check and start MySQL server
echo "Checking MySQL server status..."
if command -v systemctl &> /dev/null; then
    # Using systemd (Ubuntu/Debian/most modern Linux)
    if ! systemctl is-active --quiet mysql; then
        echo "MySQL is not running. Starting MySQL server..."
        sudo systemctl start mysql
        if [ $? -eq 0 ]; then
            echo "✓ MySQL server started successfully"
        else
            echo "✗ Failed to start MySQL server. Please start it manually."
            exit 1
        fi
    else
        echo "✓ MySQL server is already running"
    fi
elif command -v service &> /dev/null; then
    # Using service command (older systems)
    if ! service mysql status &> /dev/null; then
        echo "MySQL is not running. Starting MySQL server..."
        sudo service mysql start
        if [ $? -eq 0 ]; then
            echo "✓ MySQL server started successfully"
        else
            echo "✗ Failed to start MySQL server. Please start it manually."
            exit 1
        fi
    else
        echo "✓ MySQL server is already running"
    fi
else
    echo "⚠ Cannot detect MySQL service manager. Please ensure MySQL is running manually."
fi

echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start ASR server in background
echo "Starting ASR server on port 8100..."
python3 server.py --port 8100 &
ASR_PID=$!

# Wait a bit for ASR to start
sleep 2

# Start Translation server in background
echo "Starting Translation server on port 8200..."
python3 translation_server.py --port 8200 &
TRANSLATION_PID=$!

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              VoiceBird Local Servers Running                 ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  ASR Server:         http://localhost:8100                   ║"
echo "║  Translation Server: http://localhost:8200                   ║"
echo "║                                                              ║"
echo "║  Press Ctrl+C to stop both servers                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $ASR_PID 2>/dev/null
    kill $TRANSLATION_PID 2>/dev/null
    echo "Servers stopped."
    exit 0
}

# Trap Ctrl+C
trap cleanup INT TERM

# Wait for both processes
wait
