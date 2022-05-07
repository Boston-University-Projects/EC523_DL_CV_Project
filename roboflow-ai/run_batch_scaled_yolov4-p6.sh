#!/bin/bash -l

# Set SCC project
#$ -P dl523

# Specify hard time limit for the job. 
#   The job will be aborted if it runs longer than this time.
#   The default time is 12 hours
#$ -l h_rt=24:00:00

# Request 4 CPUs
#$ -pe omp 2

# Request 1 GPU 
#$ -l gpus=4

# Specify the minimum GPU compute capability 
#$ -l gpu_c=7

# specify gpu type
#$ -l gpu_type=V100

# specify minimum gpu memory
#$ -l gpu_memory=16G

# Send an email when the job finishes or if it is aborted (by default no email is sent), or begin
#$ -m eab

# Give job a name
#$ -N scaled_yolov4_p7_zerowaste_v16_640x640_300

# Combine output and error files into a single file
#$ -j y

# Specify the output file name
#$ -o /projectnb/dl523/projects/RWD/EC523_DL_CV_Project/roboflow-ai/log/scaled_yolov4_p7_zerowaste_v16_640x640_300_05-05-2022.qlog

# ==========> Read more about batch script files arguments, https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/

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
LOGFILE="./log"
TIMESTAMP=`date "+%Y_%m_%d-%H_%M_%S"`
PROJECT_NAME="scaled_yolov4_p7_zerowaste_v16_640x640_300"
SCC_GROUP='dl523'
echo $PROJECT_NAME\_$TIMESTAMP
DATASET_DIR='../zero-waste-10'
cd /projectnb/dl523/projects/RWD/EC523_DL_CV_Project/roboflow-ai/yolor

# If you want to do the distributed training
# python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --batch-size 16 --img 640 640 --data $DATASET_DIR/data.yaml --cfg cfg/yolov4_p7.cfg --weights '' --name yolov4_p7_$TIMESTAMP --hyp data/hyp.scratch.640.yaml --epochs 300 --augment True --sync-bn --device 0,1,2,3 --exist-ok
# or --weights weights/yolov4_p6.weights


python train.py --batch-size 16 --img 640 640 --data $DATASET_DIR/data.yaml --cfg cfg/yolov4_p7.cfg --weights weights/yolov4_p7.weights --name yolov4_p7_$TIMESTAMP --hyp data/hyp.scratch.640.yaml --epochs 300 --augment True  --device 0 --exist-ok

python test.py --conf-thres 0.0 --img 640 --batch 16 --device 0 --data $DATASET_DIR/data.yaml --cfg cfg/yolov4_p6.cfg --weights runs/train/yolov4_p6_$TIMESTAMP/weights/best_overall.pt --task test --names data/zerowaste.names --verbose --save-json --save-conf --save-txt --gt_json_dir $DATASET_DIR/test/_annotations.coco.json

# More refers to https://github.com/WongKinYiu/ScaledYOLOv4


############# (Upon completed) Managing and Tracking your Batch Job, https://www.bu.edu/tech/support/research/system-usage/running-jobs/tracking-jobs/
date
qstat -u dong760
