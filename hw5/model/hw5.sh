# $1: Food dataset directory
# $2: Output images directory
python3 saliency_main.py $1 $2
python3 filter_main.py $1 $2
python3 shap_main.py $1 $2
python3 confusion_matrix.py $1 $2
python3 Lime.py $1 $2
