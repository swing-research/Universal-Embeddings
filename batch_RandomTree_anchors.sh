# GPU 0 
nohup python -u RandomTree_anchors.py 0 0 > disp_nb_anchor0.out &
nohup python -u RandomTree_anchors.py 1 0 > disp_nb_anchor1.out &
# GPU 1 
nohup python -u RandomTree_anchors.py 2 1 > disp_nb_anchor2.out &
nohup python -u RandomTree_anchors.py 3 1 > disp_nb_anchor3.out &
# GPU 2 
nohup python -u RandomTree_anchors.py 4 2 > disp_nb_anchor4.out &
nohup python -u RandomTree_anchors.py 5 2 > disp_nb_anchor5.out &
# GPU 3
nohup python -u RandomTree_anchors.py 6 3 > disp_nb_anchor6.out &
nohup python -u RandomTree_anchors.py 7 3 > disp_nb_anchor7.out &

