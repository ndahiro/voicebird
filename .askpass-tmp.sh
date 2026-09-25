#!/bin/sh
# Temporary SSH_ASKPASS helper for the one-off password login to 192.168.0.4.
# Reads the password from the environment so it never lands in argv or on disk.
printf '%s\n' "$VB_SSH_PASSWORD"
