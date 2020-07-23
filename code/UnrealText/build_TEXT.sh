export UE_Root="/home/son97/Downloads/UnrealEngine-4.22" # your path to the UE root folder
python3 build.py --UE4 $UE_Root >> build.log.txt
more build.log.txt | grep "BUILD SUCCESSFUL"