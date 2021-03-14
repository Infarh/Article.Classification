using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;

using Microsoft.ML;
using Microsoft.ML.Data;

// ReSharper disable AsyncConverter.AsyncWait

namespace Article.Classification
{
    // https://code-ai.mk/article-classification-with-ml-net/
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();
            const string data_file_name = "urls.txt";
            var data = GetData(data_file_name).ToArray();
            var data_set = context.Data.LoadFromEnumerable(data);

            const int number_of_clusters = 3;
            var text_estimator =
                       context.Transforms.Text.NormalizeText("Text")
               .Append(context.Transforms.Text.TokenizeIntoWords("Text"))
               .Append(context.Transforms.Text.RemoveDefaultStopWords("Text"))
               .Append(context.Transforms.Conversion.MapValueToKey("Text"))
               .Append(context.Transforms.Text.ProduceNgrams("Text"))
               .Append(context.Transforms.NormalizeLpNorm("Text"))
               .Append(context.Clustering.Trainers.KMeans("Text", numberOfClusters: number_of_clusters));
            ;

            var model = text_estimator.Fit(data_set);

            var prediction_engine = context.Model.CreatePredictionEngine<TextData, Prediction>(model);

            var prediction_cpp = prediction_engine.Predict(GetDataFromUrl("https://en.wikipedia.org/wiki/C%2B%2B"));
            var prediction_java = prediction_engine.Predict(GetDataFromUrl("https://en.wikipedia.org/wiki/Java_(programming_language)"));

            var prediction_volleyball = prediction_engine.Predict(GetDataFromUrl("https://en.wikipedia.org/wiki/Volleyball"));
            var prediction_baseball = prediction_engine.Predict(GetDataFromUrl("https://en.wikipedia.org/wiki/Baseball"));

            var prediction_sleepless_in_seattle = prediction_engine.Predict(GetDataFromUrl("https://en.wikipedia.org/wiki/Sleepless_in_Seattle"));
            var prediction_gone_with_the_wind = prediction_engine.Predict(GetDataFromUrl("https://en.wikipedia.org/wiki/Gone_with_the_Wind_(film)"));
        }

        private static IEnumerable<TextData> GetData(string DataFile)
        {
            using var file = File.OpenText(DataFile);
            using var client = CreateClient();

            while (!file.EndOfStream)
            {
                if (file.ReadLine() is not { Length: > 0 } url) continue;

                Console.Write("Загрузка данных из {0}...", url);

                Console.CursorLeft -= 3;
                switch (GetDataFromUrl(url))
                {
                    case null:
                        Console.WriteLine(" ошибка!");
                        break;

                    case { Text: { Length: > 0 } } result:
                        Console.WriteLine(" выполнена успешно.");
                        yield return result;
                        break;

                    case { }:
                        Console.WriteLine(" получен пустой ответ.");
                        break;
                }
            }
        }

        private static HttpClient CreateClient(string UserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.152 YaBrowser/21.2.2.101 Yowser/2.5 Safari/537.36")
        {
            var client = new HttpClient();
            client.DefaultRequestHeaders.UserAgent.ParseAdd(UserAgent);
            return client;
        }

        private static TextData GetDataFromUrl(string url)
        {
            using var client = CreateClient();
            return GetDataFromUrl(url, client);
        }

        private static TextData GetDataFromUrl(string url, HttpClient client) =>
            client.GetAsync(url).Result is { IsSuccessStatusCode: true } response
                ? new(response.Content.ReadAsStringAsync().Result)
                : null;
    }

    public record TextData(string Text);

    public record Prediction
    {
        [ColumnName("PredictedLabel")]
        public uint Cluster { get; init; }
    }
}
