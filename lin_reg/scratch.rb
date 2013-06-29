require 'csv'
require 'matrix'
require_relative 'linear_regression'

# ./lin-reg --train filename --model modelfile
# ./lin-reg --test filename --model modelfile
# ./lin-reg --predict filename --model modelfile
# ./lin-reg --predict x1 x2 x3 x4 --model modelfile

x_mat = Matrix[[-1,8], [6,12], [-6,16], [8,15], [-8,-3]]
y_vec = Vector[11, 89, -23, 115, -81]    
lin_reg = LinearRegression.new(x_mat, y_vec)
lin_reg.prepare_data
lin_reg.train(1)
lin_reg.debug_print
xx_mat = Matrix[[100,1000], [-5,10], [-7,32], [0,4]]
puts lin_reg.predict_matrix(xx_mat).inspect
# [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30].each do |alpha|
#   begin
#     lin_reg.train(alpha)
#     cost = lin_reg.cost
#     nic = lin_reg.num_iters_till_conv
#     puts "#{alpha}, #{cost}, #{nic}"
#   rescue ArgumentError => ex
#     puts "#{alpha}, -1, -1"
#   end
# end
