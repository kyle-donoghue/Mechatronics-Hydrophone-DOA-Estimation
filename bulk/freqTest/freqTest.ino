uint16_t rawData[256];
int analogCounter;
int analogValue;
unsigned long startTime;

void setup() {
  analogCounter = 0;
  analogValue = 0;
  Serial.begin(250000);
  startTime = 0;
}

void loop() {
  analogValue = analogRead(0);
  rawData[analogCounter] = analogValue;
  analogCounter++;
  if (analogCounter == 256) {
    
    freqAnalysis(rawData,(micros()-startTime)/256);
    analogCounter = 0;
    startTime = micros();
  }
}

void freqAnalysis(uint16_t analogData[], int avgSampleTime) {
  uint16_t replication[256];
  int prevArea = 0;
  int hit = false;
  int marker = 0;
  uint16_t area[256];
  for (int i = 0; i < 256; i++) {
    replication[i] = analogData[i];
  }
  for(int counter = 0; counter < 256; counter++) {
    area[counter] = 0;
    for( int k =1; k< 256;k++) {
      int current = analogData[k]-replication[k];
      current = (current < 0) ? current*(-1) : current;
      int prev = analogData[k-1]-replication[k-1];
      prev = (prev < 0) ? prev*(-1) : prev;
      area[counter] += current+prev;
    }
    //shift data
    int temp = replication[255];
    for (int j = 255; j > 0; j--) {
      replication[j] = replication[j-1];
    }
    replication[0] = temp;
  }
  int width = local_minima(area);
  Serial.print((1000000.0/(width*avgSampleTime)));
  Serial.println(" Hz");
  return;
  
}

int local_minima(uint16_t array[]) {
  int lowestMinima = array[0];
  int lowestMarker = 0;
  for(int i = 1; i < 125;i++) {
    if(((array[i] < array[i-1])&&(array[i] < array[i+1]))) {
      if(lowestMarker == 0) {
        lowestMinima = array[i];
        lowestMarker = i;
      }
      else {
        if(lowestMinima > array[i]) {
          lowestMinima = array[i];
          lowestMarker = i;
        }
      }
      //Serial.println(i);
    }
  }
  return lowestMarker;
}
