require 'csv'
require 'matrix'
require 'narray'


# ./lin-reg --train filename --model modelfile
# ./lin-reg --test filename --model modelfile
# ./lin-reg --predict filename --model modelfile
# ./lin-reg --predict x1 x2 x3 x4 --model modelfile


class Array
  def avg
    self.reduce(:+)/self.count
  end

  def stdev
    avrg = self.avg
    s_2 = (self.reduce(0){|acc, x| acc += (x - avrg)**2})/(self.count - 1)
    Math.sqrt(s_2)
  end

  def normalize
    avrg = self.avg
    sd = self.stdev
    self.map{|x| (x - avrg)/sd}
  end
end

data_file = ARGV.shift

x_cols = [] #Array of arrays where each internal array is a column
y_col = []
CSV.foreach(data_file) do |row|
  row[0..-2].each_with_index do |e, i|
    (x_cols[i] ||= []).push(Float(e))
  end
  y_col << Float(row[-1])
end
x_cols.each_with_index do |x_col, i|
  x_cols[i] = x_col.normalize
end
$x_mat = Matrix.columns(x_cols)
$y_vec = Matrix.column_vector(y_col.normalize)
$m = $x_mat.row_size
$n = $x_mat.row(0).size

def h(theta_vec)
  $x_mat * theta_vec
end

def J(theta_vec)
  d_mat = h(theta_vec) - $y_vec
  ((d_mat.transpose)*d_mat)/(2*$m)
end

def diff_J(theta_vec, j)
  x_j_vec = Matrix.column_vector($x_mat.column(j).to_a)
  (x_j_vec.transpose * (h(theta_vec) - $y_vec))/$m
end

def new_thetas(theta_vec, alpha)
  new_theta_col = []
  theta_vec.each_with_index do |theta, j|
    new_theta_col << theta - (alpha * diff_J(theta_vec, j))[0,0]
  end
  Matrix.column_vector(new_theta_col)
end

def grad_desc(theta_vec, alpha)
  j_hist = []
  1000.times do |i|
    theta_vec = new_thetas(theta_vec, alpha)
    j_hist << J(theta_vec)[0,0]
  end
  [theta_vec, j_hist]
end

puts "\nX:"
puts $x_mat.inspect

puts "\ny:"
puts $y_vec.inspect
thetas, j_hist = grad_desc(Matrix.column_vector(Array.new($n).fill(0)), 0.01)
#puts j_hist.inspect
puts "\nWeights: "
puts thetas.inspect

puts "\nHypothesis: "
puts h(thetas)
