class DataGen(tf.keras.utils.Sequence):

    def __init__(self,
                 df_path,
                 path_sign,
                 path_pose,
                 batch_size,
                 img_size,
                 sequence_len = 60

                 ):
        self.data=[]

        label = pd.read_csv(df_path)

        self.batch_size=batch_size
        self.img_size=img_size
        self.n=label.shape[0]
        self.sequence_len=sequence_len
        self.feature_extractor = self.build_feature_extractor()
        for index, row in tqdm(label.iterrows(), total=label.shape[0]):



            #pose_path = os.path.join(path_pose, row['pose'])
            pose_path = row['pose'].replace('.pose','')
            depth_video_path = os.path.join(path_sign, row['video_depth'])
            rgb_video_path = os.path.join(path_sign, row['video_rgb'])
            
            #if ((os.path.exists(pose_path)) & (os.path.exists(rgb_video_path)) & (os.path.exists(depth_video_path))):
            sub_data={
                    'rgb_video_path':rgb_video_path,
                    'depth_video_path':depth_video_path,
                    'pose_path':pose_path,
                    'label':row['ClassId'],
            }
            self.data.append(sub_data)

        self.n=len(self.data)


    def build_feature_extractor(self):
 
        feature_type = 'clip'
        model_name = 'ViT-B/32'

        # Load and patch the config
        args = OmegaConf.load(build_cfg_path(feature_type))
        args.feature_type = feature_type
        args.device = device
        args.batch_size = 4096

        extractor = ExtractCLIP(args)


        return extractor



    def __load_data__(self,i):
        data = self.data[i]

        video = self.load_video(data['rgb_video_path'])
        depth = self.load_video(data['depth_video_path'])
        pose = self.load_pose(data['pose_path'])

        label = data['label']

        return video,depth,pose,label


    def load_pose(self,file_path):
        buffer = open(file_path, "rb").read()
        pose_data = Pose.read(buffer, NumPyPoseBody)
        pose_data.normalize_distribution()
        data = pose_data.body.data.data
        new_frame=self.sequence_len
        frame = data.shape[0]
        step = frame / new_frame
        new_matrix = np.zeros((new_frame, 135, 2))
        if(frame<=new_frame):
            for i in range(frame):
                new_matrix[i] = data[i][0]
        else:
            for i in range(new_frame):
                new_matrix[i] = data[round(i * step)][0]

        reshaped_matrix = np.reshape(new_matrix, (new_frame, 135*2))
        return  np.asarray(reshaped_matrix.astype(np.float16))

    def load_labels(self,data):
        label_processor = tf.keras.layers.StringLookup(
        num_oov_indices=0, vocabulary=np.unique(data)
        )
        label = np.asarray(label_processor(data)).astype(np.uint8)

        return label


    def crop_center_square(self,frame):
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

    def load_video(self,path):


        temp_frame_features = np.zeros(
            shape=(1, self.sequence_len, 512), dtype="float32"
        )

        new_matrix = np.zeros((self.sequence_len,512),dtype=np.float16)

        extracted = self.feature_extractor.extract(path)['clip']
        len_ext = extracted.shape[0]


        if(len_ext<=len(new_matrix)):
            for i in range(len_ext):
                  new_matrix[i] = extracted[i]
        else:
            for i in range(len(new_matrix)):
                  new_matrix[i] = extracted[round((len_ext/self.sequence_len)*i)]
        
        return new_matrix




    def __get_batch__(self,index_interval):
        i_start=index_interval[0]
        i_end=index_interval[1]
        batch_video=[]
        batch_depth=[]
        batch_pose=[]
        batch_label=[]


        for i in tqdm(range(i_start,i_end)):
            video,depth,pose,label=self.__load_data__(i)
            batch_video.append(video)
            batch_depth.append(depth)
            batch_pose.append(pose)
            batch_label.append(label)

        return np.asarray(batch_video), np.asarray(batch_depth), np.asarray(batch_pose), np.asarray(batch_label)

    def __getitem__(self, index):
        video,depth,pose,label=self.__get_batch__(index_interval=[index * self.batch_size,(index + 1) * self.batch_size])
        return video,depth,pose,label

    def __len__(self):
        return self.n // self.batch_size
