for var in 1 2 3 4 5 6 7 8 9
do
echo test_$var.png 
python use_intro.py -i test_images/test_$var.png
done