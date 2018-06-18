import os
import sys

os.system("perl ./evaluation/score_4 ./evaluation/gold/%s_dict.txt ./evaluation/gold/%s_test.txt ./output/%s_test_%s.txt > ./output/%s_%s_score.txt"\
			 % (sys.argv[1], sys.argv[2], sys.argv[2], sys.argv[3], sys.argv[2], sys.argv[3]))