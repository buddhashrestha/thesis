6mins python pyannote-structure.py shot --verbose /home/buddha/thesis/pyannote-data/TheBigBangTheory.mkv /home/buddha/thesis/pyannote-data/TheBigBangTheory.shots.json
10mins python pyannote-video/scripts/pyannote-face.py track --verbose --every=0.5 /home/buddha/thesis/pyannote-data/TheBigBangTheory.mkv /home/buddha/thesis/pyannote-data/TheBigBangTheory.shots.json pyannote-data/TheBigBangTheory.track.txt                                      
 python pyannote-video/scripts/pyannote-face.py demo /home/CS/brs0020/thesispyannote-data/TheBigBangTheory.mkv pyannote-data/TheBigBangTheory.track.txt pyannote-data/TheBigBangTheory.track.mp4               
 python pyannote-video/scripts/pyannote-face.py extract --verbose /home/CS/brs0020/thesispyannote-data/TheBigBangTheory.mkv pyannote-data/TheBigBangTheory.track.txt dlib-models/shape_predictor_68_face_landmarks.dat dlib-models/dlib_face_recognition_resnet_model_v1.dat pyannote-data/TheBigBangTheory.landmarks.txt pyannote-data/TheBigBangTheory.embedding.txt 

  python pyannote-video/scripts/pyannote-structure.py shot --verbose /home/CS/brs0020/thesis/pyannote-data/will.mp4 /home/CS/brs0020/thesis/pyannote-data/will.shots.json 
  
  
  pyannote-data/TheBigBangTheory.track.txt

  python pyannote-video/scripts/pyannote-face.py track --verbose --every=0.5 /home/CS/brs0020/thesis/pyannote-data/will.mp4 /home/CS/brs0020/thesis/pyannote-data/will.shots.json /home/CS/brs0020/thesis/pyannote-data/will.track.txt                                      
 python pyannote-video/scripts/pyannote-face.py extract --verbose /home/CS/brs0020/thesis/pyannote-data/will.mp4 pyannote-data/will.track.txt dlib-models/shape_predictor_68_face_landmarks.dat dlib-models/dlib_face_recognition_resnet_model_v1.dat pyannote-data/will.landmarks.txt pyannote-data/will.embedding.txt 


mongoimport --type csv --headerline --db mflix --collection movies_initial --host "mflix-shard-0/mflix-shard-00-00-f2wgn.mongodb.net:27017,mflix-shard-00-01-f2wgn.mongodb.net:27017,mflix-shard-00-02-f2wgn.mongodb.net:27017" --authenticationDatabase admin --ssl --username analytics --password analytics-password --file movies_initial.csv