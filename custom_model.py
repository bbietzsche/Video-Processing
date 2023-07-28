class CustomModel(tf.keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        (input_video,input_depth,input_pose) ,y = data

        with tf.GradientTape() as tape:
            y_pred = self([input_video,input_depth,input_pose] ,training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compute_loss(y=y, y_pred=y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def validation_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        (input_video,input_depth,input_pose) ,y = data

        y_pred = self([input_video,input_depth,input_pose],training=False)  # Forward pass
        # Compute the loss value
        # (the loss function is configured in `compile()`)
        loss = self.compute_loss(y=y, y_pred=y_pred)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}