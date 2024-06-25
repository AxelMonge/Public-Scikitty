import { useState } from "react";
import InitBoard from "../InitBoard";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faSpinner } from "@fortawesome/free-solid-svg-icons";
import { toast } from "sonner";

const INITIAL_BOARD_INDENTIFIER = 'ib';
const GOAL_BOARD_INDENTIFIER = 'gb';

const FormBoards = ({ boardSize, setBoardData }) => {
    const [isLoading, setIsLoading] = useState(false);

    const getRetraigoData = async (initialBoard, goalBoard) => {
        const response = await fetch('/api/retraigo', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json', 
                'cache-control': 'no-cache',
            },
            body: JSON.stringify({
                "board": initialBoard,
                "goal": goalBoard
            }),
        });

        if (!response.ok) {
            throw new Error(`Server Error: ${response.status} ${response.statusText}`);
        }
        else {
            return await response.json();
        };
    }

    const onSubmitBoard = async (e) => {
        try {
            e.preventDefault();
            setIsLoading(true);
            const formData = new FormData(e.target);
            const initialBoard = getFormBoard(formData, INITIAL_BOARD_INDENTIFIER);
            const goalBoard = getFormBoard(formData, GOAL_BOARD_INDENTIFIER);
            const boardData = await getRetraigoData(initialBoard, goalBoard);
            setBoardData(boardData);
        }
        catch (e) {
            if (e instanceof TypeError) {
                toast.error('Error!', { description: "Server not online!" });
            }
            else {
                toast.error('Error!', { description: e.message });
            };
        }
        finally {
            setIsLoading(false);
        }
    }

    const getFormBoard = (formData, identifier) => {
        const board = [];
        for (let i = 0; i < boardSize; i++) {
            const row = [];
            const valuesSet = new Set(); 
            for (let j = 0; j < boardSize; j++) {
                let value = formData.get(`${identifier}${i}${j}`);
                if (value === '0') {
                    value = 'empty';
                }
                if (valuesSet.has(value)) {
                    throw new Error(`Duplicate value found: ${value} at position (${i}, ${j})`);
                }
                valuesSet.add(value);
                row.push(value);
            }
            board.push(row);
        }
        return board;
    }
    
    return (
        <form onSubmit={onSubmitBoard}>
            <div className="flex gap-3 mt-3">
                <InitBoard boardSize={boardSize} title={'Initial Board'} identifier={INITIAL_BOARD_INDENTIFIER} />
                <InitBoard boardSize={boardSize} title={'Goal Board'} identifier={GOAL_BOARD_INDENTIFIER} />
            </div>
            <div className="flex justify-center mt-3">
                {!isLoading ? (
                    <button type="submit" className="bg-green-500 p-1 rounded w-1/3">
                        🚀 Send Boards to Play 🚀
                    </button>
                ) : (
                    <button type="submit" className="bg-green-700 p-1 rounded w-1/3 m-auto cursor-not-allowed text-gray-400">
                        <FontAwesomeIcon icon={faSpinner} className="spinner" />
                    </button>
                )}
            </div>
        </form>
    )
}

export default FormBoards;