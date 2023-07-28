
class DataGen(tf.keras.utils.Sequence):

    def __init__(self,path,batch_size,shuffle

                 ):
        self.batch_size=batch_size
        self.files=os.listdir(path)
        self.files=[os.path.join(path,x) for x in self.files if os.path.exists(os.path.join(path,x))]
        self.n=len(self.files)
        self.shuffle=shuffle



    def __load_data__(self,i):
        data=np.load(self.files[i],allow_pickle=True)[0]
        video = data['video'][0]
        depth = data['depth'][0]
        pose = data['pose'][0]
        label = data['label']

        return video,depth,pose,label

    def __get_batch__(self,index_interval):
        i_start=index_interval[0]
        i_end=index_interval[1]
        batch_video=[]
        batch_depth=[]
        batch_pose=[]
        batch_label=[]


        for i in range(i_start,i_end):
            video,depth,pose,label=self.__load_data__(i)
            batch_video.append(video)
            batch_depth.append(depth)
            batch_pose.append(pose)
            batch_label.append(label)

        return np.asarray(batch_video), np.asarray(batch_depth), np.asarray(batch_pose), np.asarray(batch_label)

    def __getitem__(self, index):
        video,depth,pose,label=self.__get_batch__(index_interval=[index * self.batch_size,(index + 1) * self.batch_size])
        return (video,depth,pose),label

    def __len__(self):
        return self.n // self.batch_size
    
    def on_epoch_end(self):
       if self.shuffle==True:
            random.shuffle(self.files)
