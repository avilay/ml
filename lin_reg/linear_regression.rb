require 'english'

class LinearRegression

  def initialize
    @x_mat = nil
    @y_vec = nil
  end

  def load_data(data_file)
    x_cols = [] #Array of arrays where each internal array is a column
    y_col = []
    CSV.foreach(data_file) do |row|      
      begin
        row[0..-2].each_with_index do |x, i|          
          (x_cols[i] ||= []).push(Float(x))
        end
        y_col << Float(row[-1])
      rescue => ex
        $logger.warn {"Cannot parse row #{row}"}
        $logger.warn {ex.message + "\n" + ex.backtrace.join("\n")}
        last_index = $INPUT_LINE_NUMBER-1
        x_cols.each do |x_col|
          x_col.delete_at(last_index)
        end
        y_col.delete_at(last_index)
      end
    end

    #normalize the data
    x_cols.each_with_index do |x_col, i|
      x_cols[i] = x_col.normalize
    end
    @x_mat = Matrix.columns(x_cols)
    @y_vec = Matrix.column_vector(y_col.normalize)
  end

  def self.train(params)
    lin_reg = LinearRegression.new
    lin_reg.load_data(params[:data])
    
    #find weights

  end

  def self.validate(params)

  end

  def self.predict(params)

  end

end