#!/bin/bash -l

LOGFILE="./log"
TIMESTAMP=`date "+%Y_%m_%d-%H_%M_%S"`
PROJECT_NAME="yolov4_db_100"
SCC_GROUP='dl523'
echo $PROJECT_NAME\_$TIMESTAMP

# Set SCC project
#$ -P dl523

# Specify hard time limit for the job. 
#   The job will be aborted if it runs longer than this time.
#   The default time is 12 hours
#$ -l h_rt=24:00:00

# Request 2 CPUs
#$ -pe omp 2

# Request 1 GPU 
#$ -l gpus=1

# Specify the minimum GPU compute capability 
#$ -l gpu_c=3.5

# specify gpu type
#$ -l gpu_type=V100

# Send an email when the job finishes or if it is aborted (by default no email is sent), or begin
#$ -m eab

# Give job a name
#$ -N yolov4_db_100

# Combine output and error files into a single file
#$ -j y

# Specify the output file name
#$ -o /projectnb/dl523/projects/RWD/logs/yolov4-db-100.qlog

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="

echo "==========================> Loading moudule for Project Environment"
module load cuda/10.1
module load python3/3.8.10


echo "===========================> Checking OS Information"
lsb_release -a
uname -m && cat /etc/*release
echo "===========================> Checking CPU Information"
lscpu
# Check you have GPU driver installed
echo "===========================> Checking GPU Configuration"
nvidia-smi

echo "==========================> Checking SCC Quota Usage"
pquota dl523
quota -s
qstat -u zgu
module list

echo "==========================> Current ENV Path"
echo $PATH
echo $PYTHONPATH
python -V
which python


echo "==========================> Start Training"
# ======================> Using Zuyu's project to train yolov4
cd /projectnb/dl523/projects/RWD/PyTorch_YOLOv4

python train.py --batch-size 16 --img-size 640 640 --data zero-waste-10/data.yaml --cfg cfg/yolov4-custom.cfg --weights weights/yolov4-csp-x-leaky.weights  --device 0 --name yolov4-db-100-ep --epochs 100 --augment True

cp runs/train/yolov4-db-100-ep/weights/best_overall.pt weights/yolov4_db_100.pt

python test.py --img-size 640 --conf-thres 0.001 --batch 16 --device 0 --data zero-waste-10/data.yaml --cfg cfg/yolov4-custom.cfg --weights weights/yolov4_db_100.pt --task test --names zero-waste-10/zero-waste.names --verbose --save-json --name yolov4_db_100

############# (Upon completed) Managing and Tracking your Batch Job, https://www.bu.edu/tech/support/research/system-usage/running-jobs/tracking-jobs/
date
qstat -u zgu