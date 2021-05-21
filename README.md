# Decongestion-time-prediction-at-Railroad-Grade-Crossing
This work presents a deep learning-assisted framework to estimate the decongestion time at the grade crossing. A hypothesis of the traffic behavior during the congestion event caused by passing trains is proposed. A deep neural network-based vehicle crowd counting algorithm is developed to estimate the number of vehicles at the normal traffic condition. A running average-based motion detection algorithm is designed to estimate the time of the train passing through the grade crossing. A regression model is then constructed to relate the quantitative information with the decongestion time. 
![Test Image 1](/images/decongestion.jpg)


We formulate a holistic methodology and framework that combines appropriately selected image analysis and deep learning approaches and quantitative mathematical models into a streamlined pipeline for decongestion time prediction at grade crossings:
![Test Image 2](/images/system.jpg)

A differential approach is proposed to address the challenge associated with data deficiency of congestion events in grade crossings. Specifically, the image data of the vehicles at the normal traffic conditions (without the train-induced congestion) is adequately available, and therefore, a deep learning-based vehicle crowd counting model is trained to evaluate the vehicle number on the road. Conversely, the train passing time within a pre-defined region of interest (ROI) is estimated by an economic motion detection-based algorithm due to limited data of congestion events.
![Test Image 3](/images/CSRNet.jpg)

It is demonstrated that a quantitative relationship between the vehicle number at the normal traffic condition and the train passing time, and the decongestion time, can be captured by a compact multivariate regression model. The study reveals that traffic congestion and decongestion at the grade crossing is indeed complex, and deciphering its behavior is an interesting research area worth further study.

![Test Image 4](/images/3D-GT.png)
