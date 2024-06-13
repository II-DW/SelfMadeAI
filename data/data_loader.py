class dataPreprocess :

    def __init__ (self, train_file_path, test_file_path) :
        self.train_label, self.train_data = self.read_line(train_file_path)
        self.test_label, self.test_data = self.read_line(test_file_path)

    def read_line(self, file_name) :
        f = open(file_name, 'r')
        n = 0
        f.readline()
        label = []
        data = []

        while True:

            line = f.readline()

            if not line: 
                break
            
            L = line.split(',')
            label.append(int(L[0]))

            y = []
            for i in range (28) :
                x = []
                for j in range (28) :
                    x.append(int(L[28*i+j+1]))
                y.append(x)
                
            data.append(y)
            if (n+1) % 10000 == 0 :
                print(file_name, "Loading...", n+1)
            n+=1

        f.close()
        return label, data
    
    def get_item (self) :
        return self.train_label, self.train_data, self.test_label, self.test_data 