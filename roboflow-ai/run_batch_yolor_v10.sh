#!/bin/bash -l
LOGFILE="./log"
TIMESTAMP=`date "+%Y_%m_%d-%H_%M_%S"`
PROJECT_NAME="yolor_p6_zerowaste_v10_150_with_updated_metrics"
SCC_GROUP='dl523'
echo $PROJECT_NAME\_$TIMESTAMP

# Set SCC project
#$ -P dl523

# Specify hard time limit for the job. 
#   The job will be aborted if it runs longer than this time.
#   The default time is 12 hours
#$ -l h_rt=24:00:00

# Request 4 CPUs
#$ -pe omp 2

# Request 1 GPU 
#$ -l gpus=2

# Specify the minimum GPU compute capability 
#$ -l gpu_c=7

# specify gpu type
#$ -l gpu_type=V100

# Send an email when the job finishes or if it is aborted (by default no email is sent), or begin
#$ -m eab

# Give job a name
#$ -N yolor_p6_zerowaste_v10_150_with_updated_metrics

# Combine output and error files into a single file
#$ -j y

# Specify the output file name
#$ -o /projectnb/dl523/projects/RWD/EC523_DL_CV_Project/roboflow-ai/log/yolor_p6_zerowaste_v10_150_with_updated_metrics_04-30-2022.qlog

# Check the metrics.py files, you will see that the best_overall defined we have changed, as follow: 
# w = [0.0, 0.0, 0.1, 0.9]  ==> w = [0.25, 0.25, 0.25, 0.25]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="

echo "==========================> Loading moudule for Project Environment"
# qrsh -P dl523 -l gpus=1 -l gpu_c=3.5
# qsub run_batch.sh
module load python3/3.8.10
# module load tensorflow/2.5.0
# module load pytorch/1.9.0
# module load opencv/4.5.0
module load cuda/11.3
# module load pandoc/2.5
# module load texlive/2018
# module load miniconda/4.9.2
# module load gcc/9.3.0
export PATH=/usr4/dl523/dong760/.local/lib/python3.8/site-packages:$PATH
export PATH=/usr4/dl523/dong760/.conda/envs/dl_env/bin:$PATH
source /projectnb/dl523/students/dong760/miniconda3/bin/activate
export PYTHONNOUSERSITE=true
conda activate dl_env


echo "===========================> Checking OS Information"
lsb_release -a
uname -m && cat /etc/*release
echo "===========================> Checking CPU Information"
lscpu
# Check you have GPU driver installed
echo "===========================> Checking GPU Configuration"
nvidia-smi
# nvidia-htop.py --color -l 30	# Read here to learn more about GPU monitoring, https://github.com/peci1/nvidia-htop
# Checking you have CUDA compiler
nvcc --version	
uname -arv

echo "==========================> Checking SCC Quota Usage"
pquota dl523
quota -s
qstat -u dong760
module list

echo "==========================> Current ENV Path"
echo $PATH
echo ""
echo $PYTHONPATH
python -V
which python


echo "==========================> Start Training"
cd /projectnb/dl523/projects/RWD/EC523_DL_CV_Project/roboflow-ai/yolor

python train.py --batch-size 16 --img 640 640 --data ../zero-waste-10/data.yaml --cfg cfg/yolor_p6.cfg --weights weights/yolor_p6.pt --device 3 --name yolor_p6_$TIMESTAMP --hyp data/hyp.scratch.1280.yaml --epochs 300 --augment True

python test.py --conf-thres 0.0 --img 640 --batch 16 --device 3 --data ../zero-waste-10/data.yaml --cfg cfg/yolor_p6.cfg --weights runs/train/yolor_p6_$TIMESTAMP/weights/best_overall.pt --task test --names data/zerowaste.names --verbose --save-json --save-conf --save-txt


# If you are running under roboflow-ai folder
# cd /projectnb/dl523/projects/RWD/EC523_DL_CV_Project/roboflow-ai

# If you are running under roboflow folder
# python ./yolor/train.py --batch-size 16 --img 640 640 --data zero-waste-10/data.yaml --cfg ./yolor/cfg/yolor_p6.cfg --weights './yolor/weights/yolor_p6.pt' --device 0 --name yolor_p6_0427 --hyp './yolor/data/hyp.scratch.1280.yaml' --epochs 50 

# python yolor/detect.py --weights "runs/train/yolor_p6_$TIMESTAMP/weights/best_overall.pt" --conf 0.0 --source zero-waste-1/test/images --names yolor/data/zerowaste.names --cfg yolor/cfg/yolor_p6.cfg

# python yolor/test.py --conf-thres 0.0 --img 640 --batch 32 --device 0 --data zero-waste-1/data.yaml --cfg yolor/cfg/yolor_p6.cfg --weights "runs/train/yolor_p6_$TIMESTAMP/weights/best_overall.pt" --task test --names yolor/data/zerowaste.names --verbose --save-json --save-conf --save-txt

# python yolor/test.py --conf 0.001 --iou 0.65 --img 640 --batch 32 --device 0 --data zero-waste-1/data.yaml --cfg yolor/cfg/yolor_p6.cfg --weights "runs/train/yolor_p6_2022_03_26-10_44_07/weights/best_overall.pt" --task test --names yolor/data/zerowaste.names --verbose --save-json --save-conf --save-txt

# python yolor/test.py --conf-thres 0.5 --img 640 --batch 32 --device cpu --data zero-waste-1/data.yaml --cfg yolor/cfg/yolor_p6.cfg --weights "runs/train/yolor_p6_2022_04_05-19_58_08/weights/best_overall.pt" --task test --names yolor/data/zerowaste.names --verbose --save-json --save-conf --save-txt

# Run under yolor folder
# cd /projectnb/dl523/students/dong760/roboflow-ai/yolor
# python test.py --conf-thres 0.5 --img 640 --batch 32 --device 0 --data ../zero-waste-10/data.yaml --cfg cfg/yolor_p6.cfg --weights ../runs/train/yolor_p6_2022_03_26-10_44_07/weights/best_overall.pt --task test --names data/zerowaste.names --verbose --save-json --save-conf --save-txt

# python train.py --batch-size 32 --img 640 640 --data ../zero-waste-10/data.yaml --cfg cfg/yolor_p6.cfg --weights weights/yolor_p6.pt --device 0 --name yolor_p6_0427 --hyp data/hyp.scratch.1280.yaml --epochs 5



# WARNING: --img-size 416 must be multiple of max stride 64, updating to 448
 

# python train.py --device 0 --batch-size 32 --img 448 448 --data zero-waste-1/data.yaml --cfg cfg/yolov4-csp-x-leaky.cfg --weights weights/yolov4-csp-x-leaky.weights --name yolov4-csp-x-leaky-300-ep --epochs 1
# python train.py --device 0 --batch-size 32 --img 448 448 --data zero-waste-4/data.yaml --cfg cfg/yolor_p6.cfg --weights weights/yolov4-csp-x-leaky.weights --name yolov4-csp-x-leaky-300-ep --epochs 1

# python -m pdb yolor/detect.py --weights "runs/train/yolor_p6_2022_04_05-19_58_08/weights/best_overall.pt" --conf 0.5 --source zero-waste-1/test/images --names ./yolor/data/zerowaste.names --cfg yolor/cfg/yolor_p6.cfg --verbose

# python yolor/test.py --conf-thres 0.001 --batch 16 --device 0 --data zero-waste-1/data.yaml --cfg cfg/yolov4-csp-x-leaky.cfg --weights weights/best.pt --task test --names zero-waste-1/zero-waste.names --verbose

############# (Upon completed) Managing and Tracking your Batch Job, https://www.bu.edu/tech/support/research/system-usage/running-jobs/tracking-jobs/
date
qstat -u dong760
