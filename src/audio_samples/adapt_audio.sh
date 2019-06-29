#Find all folders
echo 'Converting all .flac to .wav'
for D in *; do
    if [ -d "${D}" ]; then
        #echo 'folder:'
        #echo "${D}"
        # Find all subfolders
        for SubD in ./${D}/*; do
            if [ -d "${SubD}" ]; then
                #echo 'subfolder'
                #echo "${SubD}"   # your processing here
                # Find and convert all .flac to .wav 
                for old in ${SubD}/*.flac; do sox $old -r 44100 -c 1 ${SubD}/`basename $old .flac`.wav; done
            fi
        done
    fi
done

echo 'Delete all .flac'

for flac in `find . -name *flac`;do
    rm --force $flac
done


echo 'Concatenating all .wav to output.wav'
for D in *; do
    if [ -d "${D}" ]; then
        #echo 'folder:'
        #echo "${D}"
        # Find all subfolders
        for SubD in ./${D}/*; do
            if [ -d "${SubD}" ]; then
                #echo 'subfolder'
                #echo "${SubD}"   # your processing here
                # Concatenating all .wav to output.wav
                # find ./${SubD}/* -name '*.wav' -printf "file '$PWD/%p'\n"
                ffmpeg -f concat -safe 0 -i <(find ./${SubD}/* -name '*.wav' -printf "file '$PWD/%p'\n") -c copy ./${SubD}/output.wav
                echo $SubD
            fi
        done
    fi
done


for file in `find . ! -name "*output.wav" -a -name "*.wav"`;do
    rm $file
done

echo 'Splitting output.wav into 10s chunks'
for D in *; do
    if [ -d "${D}" ]; then
        #echo 'folder:'
        #echo "${D}"
        # Find all subfolders
        for SubD in ./${D}/*; do
            if [ -d "${SubD}" ]; then
                #echo 'subfolder'
                #echo "${SubD}"   # your processing here
                # Concatenating all .wav to output.wav
                # find ./${SubD}/* -name '*.wav' -printf "file '$PWD/%p'\n"
                for file in `find ./${SubD}/* -name "*output.wav"`;do
                    ffmpeg -i $file -f segment -segment_time 10 -c copy ./${SubD}/out%03d.wav
                done
                echo $SubD
            fi
        done
    fi
done

# Remove big audio 
for file in `find . -name "output.wav"`;do
    rm $file
done

# Remove transcripts
for file in `find . -name "*.txt"`;do
    rm $file
done