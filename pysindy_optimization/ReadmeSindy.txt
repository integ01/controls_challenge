Collect runs to result:
python tinyphysics.py --model_path ../models/tinyphysics.onnx --data_path ../data --num_segs 10 --controller mpcMainParams --collect
python tinyphysics.py --model_path ../models/tinyphysics.onnx --data_path ../data --num_segs 10 --controller pid --collect

Run System Identification:
python steer_modelSR3_sindy.py 
