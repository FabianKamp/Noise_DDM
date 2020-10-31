function PowFreqCorr=RunNoiseGenerator(noisecolor, range)
% function to run noise generator and saves to csv. Returns the log log 
% correlation of power and frequency
    
    % noisecolor: 'white', 'pink' or 'blue'
    % range: range of frequences in noise [lowerbound, higherbound]

keys = {'white','pink','blue'}; values = [1,2,3];
M = containers.Map(keys,values);
fs = 1000;
for k = 1:5
  disp(strcat('Running_', string(k)))
  noise = [];
  PowFreqCorr = [];
  for i = 1:1000
      temp = NoiseGenerator(4,fs,M(noisecolor),range(1),range(2),8);
      noise = [noise; temp(1000:3000)];
      [pxx, fx] = pwelch(noise(i,:),hann(100),[],[range(1):1:range(2)],fs);
      lpxx = log(pxx);
      lfx = log(fx);
      PowFreqCorr = [PowFreqCorr, corr(lfx',lpxx')];
  end 
  disp(mean(PowFreqCorr))
  csvwrite(strcat(noisecolor, '-noise_freq-', string(range(1)), '-', string(range(2)), '_', string(k), '.csv'), noise);
end
end
