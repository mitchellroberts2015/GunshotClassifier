# commands to generate models used for testing

from pyAudioAnalysis import audioTrainTest as aT

aT.featureAndTrain(["/home/atran39/seminar/TrimmedGunShots/9mmGlock","/home/atran39/seminar/TrimmedGunShots/45ACP","/home/atran39/seminar/TrimmedGunShots/AK-47","/home/atran39/seminar/TrimmedGunShots/AR-15","/home/atran39/seminar/TrimmedGunShots/Shotgun12Gauge"], 0.5, 0.5, aT.shortTermWindow, aT.shortTermStep, "svm", "classify-svm")
aT.fileClassification("/home/atran39/seminar/Gun-Shot-Sounds/.45 ACP/Colt 45 01 Singles 6TK C.M.wav", "/home/atran39/seminar/classify-svm", "svm")

aT.featureAndTrain(["/home/atran39/seminar/TrimmedGunShots/9mmGlock","/home/atran39/seminar/TrimmedGunShots/45ACP","/home/atran39/seminar/TrimmedGunShots/AK-47","/home/atran39/seminar/TrimmedGunShots/AR-15","/home/atran39/seminar/TrimmedGunShots/Shotgun12Gauge", "/home/atran39/seminar/Even More Loud Noises"], 0.5, 0.5, aT.shortTermWindow, aT.shortTermStep, "svm", "trimmed-svm")
aT.fileClassification("/home/atran39/seminar/TrimmedGunShots/AR-15/223ColtAr15Semiaut DR032101.wav", "/home/atran39/seminar/trimmed-svm", "svm")

# python pyAudioAnalysis/audioAnalysis.py classifyFolder -i TrimmedGunShots/AR-15/ --model svm --classifier general-svm --details

# works very well for real gunshots (clip of jeff, 99.1% likely)
aT.featureAndTrain(["/home/atran39/seminar/training/gunshot", "/home/atran39/seminar/training/not-gunshot"], 0.4, 0.4, aT.shortTermWindow, aT.shortTermStep, "svm", "binary-svm")
aT.fileClassification("/home/atran39/seminar/jeff-test-ak47.wav", "/home/atran39/seminar/binary-svm", "svm")

aT.featureAndTrain(["/home/atran39/seminar/training-classify/9mmGlock", "/home/atran39/seminar/training-classify/45ACP", "/home/atran39/seminar/training-classify/AK-47", "/home/atran39/seminar/training-classify/AR-15", "/home/atran39/seminar/training-classify/Shotgun12Gauge", "/home/atran39/seminar/training/not-gunshot"], 0.4, 0.4, aT.shortTermWindow, aT.shortTermStep, "svm", "classifier-svm")
aT.fileClassification("/home/atran39/seminar/jeff-test-ak47.wav", "/home/atran39/seminar/classifier-svm", "svm")

# shrinking window appears to fix doublbletap bug
# wecho messes up

# python audioAnalysis.py classifyFolder -i /home/atran39/seminar/testing/gunshots/ --model svm --classifier /home/atran39/seminar/binary-svm --details
# python audioAnalysis.py classifyFolder -i /home/atran39/seminar/testing/non-gunshots/ --model svm --classifier /home/atran39/seminar/binary-svm --details
