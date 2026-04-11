#!/bin/bash
# Script de reinicio completo de PadelStats

PROJECT_DIR="/Users/miguelizaga/Projects/paddelstats"

echo "🛑 Deteniendo servicios..."
lsof -ti:8001 | xargs kill -9 2>/dev/null && echo "  ✓ Backend detenido" || echo "  ℹ️  Backend no estaba corriendo"
lsof -ti:3000 | xargs kill -9 2>/dev/null && echo "  ✓ Frontend detenido" || echo "  ℹ️  Frontend no estaba corriendo"
sleep 2

echo ""
echo "🚀 Iniciando backend en puerto 8001..."
cd "$PROJECT_DIR"
export TRACKNET_DIR="$PROJECT_DIR/backend/external/TrackNetV3"
export TRACKNET_MODEL="$PROJECT_DIR/backend/external/tracknet_weights/TrackNet_best.pt"
export INPAINTNET_MODEL="$PROJECT_DIR/backend/external/tracknet_weights/InpaintNet_best.pt"
backend/venv/bin/python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001 > backend.log 2>&1 &
BACKEND_PID=$!
sleep 3

# Verificar que el backend arrancó
if lsof -ti:8001 > /dev/null 2>&1; then
    echo "  ✅ Backend corriendo (PID: $BACKEND_PID)"
else
    echo "  ❌ Error al iniciar backend. Ver backend.log"
    exit 1
fi

echo ""
echo "🚀 Iniciando frontend en puerto 3000..."
cd "$PROJECT_DIR/frontend"
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
sleep 4

# Verificar que el frontend arrancó
if lsof -ti:3000 > /dev/null 2>&1; then
    echo "  ✅ Frontend corriendo (PID: $FRONTEND_PID)"
else
    echo "  ❌ Error al iniciar frontend. Ver frontend.log"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Servicios iniciados correctamente"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  🔹 Backend:  http://localhost:8001"
echo "  🔹 Frontend: http://localhost:3000"
echo ""
echo "Para ver logs en tiempo real:"
echo "  tail -f $PROJECT_DIR/backend.log"
echo "  tail -f $PROJECT_DIR/frontend.log"
echo ""
