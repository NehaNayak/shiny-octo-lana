cat Phase1Answers-1* | tr ':' '\n' | tr 'A-Z' 'a-z' | tr -dc 'a-z\n\t' | sort | uniq > vocab.txt
cat Phase1Answers-1a.txt Phase1Answers-1b.txt | tr ':' '\t' | tr 'A-Z' 'a-z' | tr -dc 'a-z\n\t' | awk '{print $2"\t"$1}' > All_ab.txt
cat Phase1Answers-1c.txt Phase1Answers-1d.txt | tr ':' '\t' | tr 'A-Z' 'a-z' | tr -dc 'a-z\n\t' | awk '{print $2"\t"$1}' > All_cd.txt

# TrainTest separation
sort -R All_ab.txt > temp
head -n50 temp > ab_Train.txt 
tail -n32 temp > ab_Test.txt 
mv All_ab.txt ab_All.txt
mv All_cd.txt cd_All.txt
sort -R cd_All.txt > temp
head -n50 temp > cd_Train.txt
tail -n35 temp > cd_Test.txt
