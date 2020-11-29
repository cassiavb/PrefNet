#!/bin/bash

#SERVERDIR=/path/on/GPU/machine/to/dc_tts_osw/   ## change this to e.g. /disk/scratch/.../dc_tts_osw
#SOURCE=/path/on/AFS/to/dc_tts_osw/  ## e.g. in your AFS home directory

SERVERDIR=/disk/scratch/cvbotinh/PrefNet/
SOURCE=/afs/inf.ed.ac.uk/group/cstr/projects/project_tts/PrefNet/scripts/

rsync -avzh --progress $SOURCE/ $SERVERDIR/ 

