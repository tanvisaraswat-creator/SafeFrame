#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh  ·  SafeFrame Quick-Start Script
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo ""
echo "🛡️  SafeFrame · AI Content Moderation System"
echo "─────────────────────────────────────────────"

# 1. Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo "→ Creating virtual environment …"
  python3 -m venv venv
fi

# 2. Activate
source venv/bin/activate

# 3. Install dependencies
echo "→ Installing dependencies …"
pip install --quiet -r requirements.txt

# 4. Create runtime dirs
mkdir -p static/uploads logs

# 5. Launch
echo ""
echo "✅ Starting server at http://localhost:5000"
echo "   Upload UI  →  http://localhost:5000"
echo "   Admin      →  http://localhost:5000/admin"
echo ""
python app.py
