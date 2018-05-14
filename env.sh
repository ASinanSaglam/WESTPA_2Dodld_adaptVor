#export WEST_ROOT=../../../

if [[ -z "$WEST_ROOT" ]]; then
    echo "Must set environ variable WEST_ROOT"
    exit
fi

export WEST_PYTHON=$(which python2.7)
export WM_WORK_MANAGER=serial
export WEST_SIM_ROOT=/home/boltzmann/Development/test_adaptiveVoronoi/odld_enc
