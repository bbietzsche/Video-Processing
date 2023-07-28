def get_sequence_model():


    input_depth =tf.keras.layers.Input((60, 512))
    input_video = tf.keras.layers.Input((60, 512))
    input_pose = tf.keras.layers.Input((60, 270))
    
    pose = tf.keras.layers.Dense(256, activation="relu")(input_pose)
    pose = tf.keras.layers.Dropout(0.4)(pose)
    pose = tf.keras.layers.Dense(512, activation="relu")(pose)

    depth  = tf.keras.layers.Dense(256, activation="relu")(input_depth)
    depth  = tf.keras.layers.Dropout(0.4)(depth)
    depth  = tf.keras.layers.Dense(512, activation="relu")(depth )
    
    video = tf.keras.layers.Dense(256, activation="relu")(input_video)
    video = tf.keras.layers.Dropout(0.4)(video)
    video = tf.keras.layers.Dense(512, activation="relu")(video)
    
    
    x = tf.keras.layers.Concatenate(axis=-1)([input_video,input_depth,pose])
    
    x = tf.keras.layers.GRU(128, return_sequences=True)(
        x
    )
    x = tf.keras.layers.GRU(256)(x)


    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    
    output = tf.keras.layers.Dense(226, activation="softmax")(x)

    rnn_model = CustomModel([input_video,input_depth,input_pose], output)
    rnn_model.summary()

    return rnn_model