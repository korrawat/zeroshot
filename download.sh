username=suchanv
accesskey=4acfc85e7f28c334748b57594c454d7fc3e99b38
download_dir="/Volumes/Kritkorn/imagenet"
count=1

mkdir -p $download_dir

while read wnid ; do
    if [ -f "$download_dir/$wnid.tar" ] || [ -d "$download_dir/$wnid" ] ; then
        echo "Skip $wnid (count: $count); already exists"
    else
        echo "Start downloading $wnid (count: $count)"
        url="http://www.image-net.org/download/synset?wnid=$wnid&username=$username&accesskey=$accesskey&release=latest&src=stanford"
        time wget -O "$download_dir/$wnid.tar" $url
        echo "Finished downloading $wnid (count: $count)"
    fi

    if [ ! -d "$download_dir/$wnid" ] ; then
        mkdir "$download_dir/$wnid"
        tar -xf "$download_dir/$wnid.tar" -C "$download_dir/$wnid"
        rm "$download_dir/$wnid.tar"
    fi
    (( count++ ))
done < available_hop2.txt