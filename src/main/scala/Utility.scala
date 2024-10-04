import org.apache.hadoop.io.Text

import scala.jdk.CollectionConverters.*

object Utility {
    def calculateAverage(iterator: java.util.Iterator[Text]): Array[Float] = {
        val list = iterator.asScala.toList
        val sumArray = list.foldLeft(Array.fill(5)(0f)) { (acc, text) =>
            val parsedArray = parseArray(text)
            acc.zip(parsedArray).map { case (sum, value) => sum + value }
        }

        // Calculate the average for each element
        sumArray.map(_ / list.length)
    }

    // Convert Text to String and parse it to Array[Float]
    def parseArray(text: Text): Array[Float] = {
        text.toString
          .replace("[", "") // Remove the opening bracket
          .replace("]", "") // Remove the closing bracket
          .split(",") // Split by comma
          .map(_.trim.toFloat) // Trim whitespace and convert to Float
    }
}
