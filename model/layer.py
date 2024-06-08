class Conv2d  :
    def __init__ (self, input_x, input_y, kernal_x, kernal_y, stride) :
        self.input_size = (input_x, input_y)
        self.kernal_size = (kernal_x, kernal_y)
        self.stride = stride