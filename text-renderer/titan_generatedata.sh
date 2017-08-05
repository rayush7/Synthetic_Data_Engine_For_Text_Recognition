#!/bin/bash
#$ -cwd
echo "Starting"

source /etc/profile.d/modules.sh
module add apps/python/2.7.3/gcc-4.4.6
module add apps/cmake/2.8.9/gcc-4.4.6
module add apps/setuptools/0.6c11/python-2.7.3
module add apps/cython/0.18/gcc-4.4.6+python-2.7.3
# NUMPY HAS PROBLEMS WITH numpy.dot()
module add libs/numpy/1.6.2/gcc-4.4.6+python-2.7.3+atlas-3.10.0

# TEMPTING TO USE FOLLOWING:
#module add libs/pil/1.1.7/gcc-4.4.6+python-2.7.3
# BUT THAT DOESNT WORK SO USE:
export PYTHONPATH=/users/max/Programs/commonPy/lib/python/PIL:$PYTHONPATH

source /users/max/Work/virtual_envs/paintings_env/bin/activate

cd /users/max/Work/text-renderer

export ISTITAN=1

echo "Finished setup"

python generate_word_training_data.py $1