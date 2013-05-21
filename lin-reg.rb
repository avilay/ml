# ./lin-reg --train filename --model modelfile
# ./lin-reg --validate filename --model modelfile
# ./lin-reg --predict filename --model modelfile
# ./lin-reg --predict x1 x2 x3 x4 --model modelfile

require 'csv'
require 'logger'
require 'optparse'
require 'ostruct'

options = OpenStruct.new
opts = OptionParser.new do |opts|
  blurb_banner = <<-EOS
Usage: lin-reg [-t|-v|-p [x1,x2,x3]] [-d DATAFILE] -m MODELFILE
--train --data DATAFILE --model MODELFILE to train (and test) on DATAFILE
--validate --data DATAFILE --model MODELFILE to validate existing model on DATAFILE
--predict x1,x2,x3,... --model MODELFILE to predict a single instance
--predict --data DATAFILE --model MODELFILE to predict all instances in DATAFILE
  EOS
  opts.banner = blurb_banner
  opts.separator ""
  opts.separator "Options:"
  
  opts.on('-t', '--train', "Train and test with data in DATAFILE.") do
    options.train = true
  end

  opts.on('-v', '--validate', 'Validate with data in DATAFILE.') do
    options.validate = true    
  end

  opts.on('-d', '--data DATAFILE', 'Dataset for training, validation, or prediction.') do |df|
    options.data_file = df
  end

  blurb_model = "The MODELFILE is the output for training and input for validating and prediction."
  opts.on('-m', '--model MODELFILE', blurb_model) do |mf|
    options.model_file = mf
  end

  blurb_predict = "Predict instances on the command line or in DATAFILE."
  opts.on('-p', '--predict [x1,x2,x3]', Array, blurb_predict) do |list|
    options.predict = true
    options.instance = list
  end

  opts.on_tail("-h", "--help", "Show this message") do
    puts opts
    exit
  end
end

unless options.model_file
  puts "Please specify the model file.\n" + opts.to_s
  exit
end

if options.train && options.data_file
  LinearRegression.train(data: options.data_file, model: options.model_file)
elsif options.validate && options.data_file
  LinearRegression.validate(data: options.data_file, model: options.model_file)
elsif options.predict && options.data_file
  LinearRegression.predict(instances: options.data_file, model: options.model_file, ostream: $stdout)
elsif options.predict && options.instance
  LinearRegression.predict(instance: options.instance, model: options.model_file, ostream: $stdout)
else
  puts "Please supply a valid combination of options."
  puts opts.to_s
end
