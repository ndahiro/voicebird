#!/bin/bash

# Fix home directory permissions script
# Run this ON the server after SSH

set -e

echo "🔧 Fixing home directory permissions..."

# Check current user
CURRENT_USER=$(whoami)
echo "Current user: $CURRENT_USER"

# Check if we're in sudo group
if groups | grep -q sudo; then
    echo "✓ User is in sudo group"
else
    echo "✗ User is NOT in sudo group - cannot fix permissions"
    exit 1
fi

# Try to fix home directory ownership
echo "Changing ownership of /home/$CURRENT_USER..."
sudo chown -R $CURRENT_USER:$CURRENT_USER /home/$CURRENT_USER

# Verify the change
echo "Verifying permissions:"
ls -ld /home/$CURRENT_USER

# Try to move voicebird folder if it exists in /tmp
if [ -d "/tmp/voicebird" ]; then
    echo "Moving /tmp/voicebird to home directory..."
    mv /tmp/voicebird /home/$CURRENT_USER/
    echo "✓ VoiceBird moved to /home/$CURRENT_USER/voicebird"
else
    echo "Note: /tmp/voicebird not found"
fi

echo "✅ Home directory permissions fixed!"
