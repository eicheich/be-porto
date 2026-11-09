#!/bin/bash
# Script update / post-pull otomatis di cPanel
set -e

echo "[1/2] Menarik perubahan terbaru dari repository Git..."
git pull origin main

echo "[2/2] Memicu reload aplikasi Phusion Passenger..."
mkdir -p tmp
touch tmp/restart.txt

echo "[SUCCESS] Aplikasi berhasil diperbarui dan direload!"

