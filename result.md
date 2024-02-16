## FER GiMeFive 
1. GiMeFiveRes (see model.py)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
#95051014
- Train: 95.2
- Test: 70.4
- Valid: 74.5

2. GiMeFive (result_ec2DO.csv)

#10478086
- Train: 94.6
- Test: 72.5
- Valid: 73.5

3. ResNet18 (result_adam_res18.csv)

optimizer = optim.Adam(model.parameters(), lr=0.001)
#11179590
- Train: 96.5
- Test: 72.4
- Valid: 73.8

4. VGG16 (result_vgg16_SGD.csv)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
#72460742

- Train: 79.4
- Test: 53.7
- Valid: 61.4

### On RAF-DB
- Train: 93.8
- Test: 83.3
- Valid: 68.8

### On FER2013

5. 
- Train: 69.0
- Test: 49.4
- Valid: 35.6

## TODO

add model.summary()

add confusion matrix