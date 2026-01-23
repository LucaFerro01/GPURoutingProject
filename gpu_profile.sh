#!/bin/bash 
#SBATCH --job-name=GPU_routing_project 
#SBATCH --partition=gpu_v100 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4 
#SBATCH --gres=gpu:4 
#SBATCH --cpus-per-task=8 
#SBATCH --time=00:30:00 
#SBATCH --mem=16G 
#SBATCH --output=routing_profile.log 
 
PROFILE_TOOL="nsys"
APP="gpu_routing"
 
NUM_PACKETS=10000000
NUM_ROUTES=100
 
NSYS_TRACE_MPI="cuda,osrt,mpi,nvtx" 
NCU_SECTIONS="SpeedOfLight,MemoryWorkloadAnalysis,LaunchStats,Occupancy" 
 
module purge 
module load gcc/12.4.0 
module load nvhpc/25.1 
module load openmpi/5.0.7_gcc12
 
if [ -z "${CUDA_HOME:-}" ]; then 
  if [ -n "${NVHPC:-}" ]; then 
    CUDA_HOME="$NVHPC/Linux_x86_64/25.1/cuda" 
  elif command -v nvcc >/dev/null 2>&1; then 
    CUDA_HOME="$(dirname $(dirname $(command -v nvcc)))" 
  fi 
fi 
export PATH="$CUDA_HOME/bin:$PATH" 
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

PROJECT_DIR=$(pwd)
EXECUTABLE="${PROJECT_DIR}/gpu_router"

if [ ! -f "$EXECUTABLE" ]; then
  echo "ERROR: $EXECUTABLE not found!"
  exit 1
fi

echo "=== Starting profiling ==="
echo "Packets: $NUM_PACKETS, Routes: $NUM_ROUTES"
echo ""

if [ "$PROFILE_TOOL" = "nsys" ]; then 
  # Profiling con nsys
  nsys profile --force-overwrite=true \
    -o "${PROJECT_DIR}/${APP}_nsys" \
    --trace="$NSYS_TRACE_MPI" \
    --sample=cpu \
    srun -n4 --cpu-bind=cores "$EXECUTABLE" $NUM_PACKETS $NUM_ROUTES
  
  echo ""
  echo "=== Profiling completed, checking files ==="
  
  # Verifica quali file sono stati generati
  ls -lh "${PROJECT_DIR}/${APP}_nsys"* 2>/dev/null || echo "No nsys files found!"
  
  # Prova a generare stats dai file disponibili
  if [ -f "${PROJECT_DIR}/${APP}_nsys.nsys-rep" ]; then
    echo ""
    echo "=== Generating stats from .nsys-rep ==="
    nsys stats "${PROJECT_DIR}/${APP}_nsys.nsys-rep" > "${PROJECT_DIR}/${APP}_nsys_summary.txt" 2>&1
    
    if [ $? -eq 0 ]; then
      echo "SUCCESS: Summary generated in ${APP}_nsys_summary.txt"
      echo ""
      echo "=== First 50 lines of summary ==="
      head -50 "${PROJECT_DIR}/${APP}_nsys_summary.txt"
    else
      echo "WARNING: nsys stats failed, but you can still open the .nsys-rep file in Nsight Systems GUI"
    fi
  fi
  
  if [ -f "${PROJECT_DIR}/${APP}_nsys.qdrep" ]; then
    echo ""
    echo "=== Trying stats from .qdrep ==="
    nsys stats "${PROJECT_DIR}/${APP}_nsys.qdrep" > "${PROJECT_DIR}/${APP}_nsys_qdrep_summary.txt" 2>&1
  fi
  
  echo ""
  echo "=== Generated files ==="
  ls -lh "${PROJECT_DIR}/${APP}_nsys"*
  echo ""
  echo "Main report for GUI: ${APP}_nsys.nsys-rep"
  
elif [ "$PROFILE_TOOL" = "ncu" ]; then 
  echo "Running with Nsight Compute (rank 0 only)..."
  ncu --set full \
    --section "$NCU_SECTIONS" \
    --export "${PROJECT_DIR}/${APP}_ncu" \
    srun -n1 "$EXECUTABLE" $NUM_PACKETS $NUM_ROUTES
  
  echo "Generated: ${APP}_ncu.ncu-rep"
fi

echo ""
echo "=== Profiling complete ==="
