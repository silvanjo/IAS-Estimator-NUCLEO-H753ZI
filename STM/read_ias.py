import serial
import csv
import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Read IAS values from STM32 over UART")
    parser.add_argument("port", help="Serial port (e.g. COM3, /dev/ttyACM0)")
    parser.add_argument("-b", "--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("-o", "--output", default=None, help="Output CSV file (default: ias_<timestamp>.csv)")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"ias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    ser = serial.Serial(args.port, args.baud, timeout=1)
    print(f"Listening on {args.port} at {args.baud} baud...")
    print(f"Writing to {args.output}")
    print("Press Ctrl+C to stop.\n")

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_ms", "ias_hz", "frame_rms"])

        try:
            while True:
                line = ser.readline().decode("ascii", errors="ignore").strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) != 3:
                    continue

                try:
                    time_ms = int(parts[0])
                    ias_hz = float(parts[1])
                    frame_rms = float(parts[2])
                except ValueError:
                    continue

                writer.writerow([time_ms, ias_hz, frame_rms])
                f.flush()
                print(f"t={time_ms:>8d} ms  IAS={ias_hz:6.2f} Hz  RMS={frame_rms:.4f}")
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            ser.close()


if __name__ == "__main__":
    main()
