# Task2: in terminal, type the following code will initiate training and predicting process:

#python train.py --data_dir 'flowers' --save_dir 'checkpoint_vgg16_191006.pth' --arch "vgg16" --learning_rate 0.001 --input_unit 25088 --hidden_unit 4096  --class_number 102 --epochs 4 --gpu --dropout 0.2

#python predict.py --checkpoint_dir 'checkpoint_vgg16_191006.pth' --test_image_path 'flowers/test/11/image_03141.jpg' --gpu --topk 5