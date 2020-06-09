# $1: test.npy
# $2: model
# $3: prediction.csv
if [[ $2 == *"best"* ]]; then
    echo "Using model $2"
    python3 hw10_test_best.py $1 $2 $3
elif [[ $2 == *"baseline"* ]]; then
    echo "Using BASE model"
    python3 hw10_test_baseline.py $1 $2 $3
fi
