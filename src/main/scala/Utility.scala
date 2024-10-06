import org.apache.hadoop.io.Text

import scala.jdk.CollectionConverters.*

/**
 * Utility object that provides helper functions to process and calculate averages from 
 * Hadoop MapReduce input values. The primary functions include parsing arrays from Text 
 * and calculating averages across multiple arrays.
 */
object Utility {

    /**
     * Calculates the average of arrays parsed from a list of `Text` objects.
     * Each `Text` object is expected to contain a comma-separated list of floats in 
     * the format: "[x1, x2, x3, x4, x5]". This function parses those arrays, sums their values, 
     * and computes the element-wise average.
     *
     * @param iterator An iterator over `Text` values containing comma-separated arrays.
     * @return An array of `Float` values representing the average of the parsed arrays.
     */
    def calculateAverage(iterator: java.util.Iterator[Text]): Array[Float] = {
        val list = iterator.asScala.toList
        val sumArray = list.foldLeft(Array.fill(5)(0f)) { (acc, text) =>
            val parsedArray = parseArray(text)
            acc.zip(parsedArray).map { case (sum, value) => sum + value }
        }

        // Calculate the average for each element
        sumArray.map(_ / list.length)
    }

    /**
     * Parses a `Text` object into an array of `Float` values. 
     * The input `Text` should be formatted as a comma-separated list of floats
     * enclosed in square brackets (e.g., "[1.0, 2.0, 3.0, 4.0, 5.0]").
     * This function removes the brackets, splits the text by commas, and converts 
     * each element into a `Float`.
     *
     * @param text The `Text` object to be parsed.
     * @return An array of `Float` values parsed from the input text.
     */
    private def parseArray(text: Text): Array[Float] = {
        text.toString
          .replace("[", "") // Remove the opening bracket
          .replace("]", "") // Remove the closing bracket
          .split(",") // Split by comma
          .map(_.trim.toFloat) // Trim whitespace and convert to Float
    }
}
