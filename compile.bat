@echo off
cd src
cl Main.cpp NNet.cpp Util.cpp Trainer.cpp Batcher.cpp
move Main.exe ../Main.exe
cd ..