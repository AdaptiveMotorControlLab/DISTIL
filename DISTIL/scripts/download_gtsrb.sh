# please download the following files and put them in . folder
cd ..
mkdir data
cd data
pwd
wget -P . https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip --no-check-certificate
wget -P . https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip --no-check-certificate
wget -P . https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip --no-check-certificate
mkdir ./gtsrb;
mkdir ./gtsrb/Train;
mkdir ./gtsrb/Test;
mkdir ./temps;
unzip ./GTSRB_Final_Training_Images.zip -d ./temps/Train;
unzip ./GTSRB_Final_Test_Images.zip -d ./temps/Test;
mv ./temps/Train/GTSRB/Final_Training/Images/* ./gtsrb/Train;
mv ./temps/Test/GTSRB/Final_Test/Images/* ./gtsrb/Test;
unzip ./GTSRB_Final_Test_GT.zip -d ./gtsrb/Test/;
rm -r ./temps;
rm ./*.zip;
echo "Download Completed";
