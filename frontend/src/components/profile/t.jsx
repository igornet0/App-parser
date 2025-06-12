//       const ApexChart = () => {
//         const [state, setState] = React.useState({
          
//             series: [{
//               name: 'candle',
//               data: [
//                 {
//                   x: new Date(1538778600000),
//                   y: [6629.81, 6650.5, 6623.04, 6633.33]
//                 },
                
//               ]
//             }],
//             options: {
//               chart: {
//                 height: 350,
//                 type: 'candlestick',
//               },
//               title: {
//                 text: 'CandleStick Chart - Category X-axis',
//                 align: 'left'
//               },
//               annotations: {
//                 xaxis: [
//                   {
//                     x: 'Oct 06 14:00',
//                     borderColor: '#00E396',
//                     label: {
//                       borderColor: '#00E396',
//                       style: {
//                         fontSize: '12px',
//                         color: '#fff',
//                         background: '#00E396'
//                       },
//                       orientation: 'horizontal',
//                       offsetY: 7,
//                       text: 'Annotation Test'
//                     }
//                   }
//                 ]
//               },
//               tooltip: {
//                 enabled: true,
//               },
//               xaxis: {
//                 type: 'category',
//                 labels: {
//                   formatter: function(val) {
//                     return dayjs(val).format('MMM DD HH:mm')
//                   }
//                 }
//               },
//               yaxis: {
//                 tooltip: {
//                   enabled: true
//                 }
//               }
//             },
          
          
//         });

        

//         return (
//           <div>
//             <div id="chart">
//                 <ReactApexChart options={state.options} series={state.series} type="candlestick" height={350} />
//               </div>
//             <div id="html-dist"></div>
//           </div>
//         );
//       }

//       const domContainer = document.querySelector('#app');
//       ReactDOM.render(<ApexChart />, domContainer);







//   const options = {
//     responsive: true,
//     maintainAspectRatio: false,
//     scales: {
//       x: {
//         type: 'time',
//         time: {
//           unit: timeframe === '1y' ? 'month' : 
//                 timeframe === '1M' ? 'week' : 
//                 timeframe === '1w' ? 'day' : 
//                 timeframe === '1d' ? 'hour' : 'minute',
//           tooltipFormat: 'dd MMM yyyy HH:mm',
//         },
//         grid: {
//           display: false,
//         },
//       },
//       y: {
//         suggestedMin: currentPriceData ? currentPriceData.price * 0.99 : 0,
//         suggestedMax: currentPriceData ? currentPriceData.price * 1.01 : 1000,
//         position: 'right',
//         ticks: {
//           callback: function(value) {
//             return '$' + value.toLocaleString('ru-RU', { minimumFractionDigits: 2, maximumFractionDigits: 5 });
//           }
//         },
//         grid: {
//           color: 'rgba(0, 0, 0, 0.05)',
//         },
//       },
//       volume: {
//         position: 'left',
//         display: false,
//         grid: {
//           display: false,
//         },
//       },
//     },
//     plugins: {
//       legend: {
//         display: false,
//       },
//       tooltip: {
//         callbacks: {
//           label: function(context) {
//             const point = context.raw;
            
//             if (context.datasetIndex === 0) {
//               const open = point.o;
//               const high = point.h;
//               const low = point.l;
//               const close = point.c;
//               const change = close - open;
//               const changePercent = (change / open) * 100;

//               // Форматируем значения
//               const formattedOpen = formatNumber(open);
//               const formattedHigh = formatNumber(high);
//               const formattedLow = formatNumber(low);
//               const formattedClose = formatNumber(close);
//               const formattedChange = formatNumber(Math.abs(change));
//               const formattedChangePercent = Math.abs(changePercent).toFixed(2);

//               // Определяем знак для изменения
//               const changeSign = change < 0 ? '–' : '';
//               const changePercentSign = changePercent < 0 ? '–' : '';

//               return [
//                 `open: ${formattedOpen}`,
//                 `max: ${formattedHigh}`,
//                 `min: ${formattedLow}`,
//                 `close: ${formattedClose}`,
//                 `change: ${changeSign}${formattedChange} (${changePercentSign}${formattedChangePercent}%)`
//               ];
//             } else {
//               return `Объём: ${formatVolume(point.y)}`;
//             }
//           },
//           title: function(context) {
//             const date = new Date(context[0].raw.x);
//             return date.toLocaleTimeString('ru-RU', { 
//               hour: '2-digit', 
//               minute: '2-digit',
//               day: '2-digit',
//               month: 'short',
//               year: 'numeric'
//             });
//           }
//         },
//         displayColors: false,
//         backgroundColor: 'rgba(0, 0, 0, 0.8)',
//         padding: 12,
//         bodyFont: {
//           family: 'monospace',
//           size: 13
//         },
//         zoom: {
//             zoom: {
//                 wheel: {
//                 enabled: true,
//                 },
//                 pinch: {
//                 enabled: true
//                 },
//                 mode: 'xy',
//             }
//         }
//       }
//     },
//     interaction: {
//       mode: 'index',
//       intersect: false,
//     },
//   };


// //     datasets: [
// //       {
// //         label: selectedCoin,
// //         type: 'candlestick',
// //         data: chartData || [],
// //         color: {
// //           up: 'rgba(39, 174, 96, 1)',
// //           down: 'rgba(231, 76, 60, 1)',
// //           unchanged: 'rgba(0, 0, 0, 1)',
// //         },
// //         borderColor: (ctx) => {
// //           const point = ctx.raw;
// //           return point.c >= point.o 
// //             ? 'rgba(39, 174, 96, 1)' 
// //             : 'rgba(231, 76, 60, 1)';
// //         },
// //         borderWidth: 1,
// //       },
// //       {
// //         type: 'bar',
// //         label: 'volume',
// //         data: volumeData,
// //         backgroundColor: (ctx) => {
// //           const index = ctx.dataIndex;
// //           const candle = chartData?.[index];
// //           return candle && candle.c >= candle.o 
// //             ? 'rgba(39, 174, 96, 0.4)' 
// //             : 'rgba(231, 76, 60, 0.4)';
// //         },
// //         yAxisID: 'volume',
// //         borderWidth: 0,
// //       }
// //     ],
// //   };    



//       const ApexChart = () => {
//         const [state, setState] = React.useState({
          
//             series: [{
//               name: 'Income',
//               type: 'column',
//               data: [1.4, 2, 2.5, 1.5, 2.5, 2.8, 3.8, 4.6]
//             }, {
//               name: 'Cashflow',
//               type: 'column',
//               data: [1.1, 3, 3.1, 4, 4.1, 4.9, 6.5, 8.5]
//             }, {
//               name: 'Revenue',
//               type: 'line',
//               data: [20, 29, 37, 36, 44, 45, 50, 58]
//             }],


            
//             options: {
//               chart: {
//                 height: 350,
//                 type: 'line',
//                 stacked: false
//               },
//               dataLabels: {
//                 enabled: false
//               },
//               stroke: {
//                 width: [1, 1, 4]
//               },
//               title: {
//                 text: 'XYZ - Stock Analysis (2009 - 2016)',
//                 align: 'left',
//                 offsetX: 110
//               },
//               xaxis: {
//                 categories: [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016],
//               },
//               yaxis: [
//                 {
//                   seriesName: 'Income',
//                   axisTicks: {
//                     show: true,
//                   },
//                   axisBorder: {
//                     show: true,
//                     color: '#008FFB'
//                   },
//                   labels: {
//                     style: {
//                       colors: '#008FFB',
//                     }
//                   },
//                   title: {
//                     text: "Income (thousand crores)",
//                     style: {
//                       color: '#008FFB',
//                     }
//                   },
//                   tooltip: {
//                     enabled: true
//                   }
//                 },
//                 {
//                   seriesName: 'Cashflow',
//                   opposite: true,
//                   axisTicks: {
//                     show: true,
//                   },
//                   axisBorder: {
//                     show: true,
//                     color: '#00E396'
//                   },
//                   labels: {
//                     style: {
//                       colors: '#00E396',
//                     }
//                   },
//                   title: {
//                     text: "Operating Cashflow (thousand crores)",
//                     style: {
//                       color: '#00E396',
//                     }
//                   },
//                 },
//                 {
//                   seriesName: 'Revenue',
//                   opposite: true,
//                   axisTicks: {
//                     show: true,
//                   },
//                   axisBorder: {
//                     show: true,
//                     color: '#FEB019'
//                   },
//                   labels: {
//                     style: {
//                       colors: '#FEB019',
//                     },
//                   },
//                   title: {
//                     text: "Revenue (thousand crores)",
//                     style: {
//                       color: '#FEB019',
//                     }
//                   }
//                 },
//               ],
//               tooltip: {
//                 fixed: {
//                   enabled: true,
//                   position: 'topLeft', // topRight, topLeft, bottomRight, bottomLeft
//                   offsetY: 30,
//                   offsetX: 60
//                 },
//               },
//               legend: {
//                 horizontalAlign: 'left',
//                 offsetX: 40
//               }
//             },
          
          
//         });